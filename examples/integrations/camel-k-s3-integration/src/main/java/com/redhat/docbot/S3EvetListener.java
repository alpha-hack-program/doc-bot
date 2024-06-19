package com.redhat.docbot;

import org.apache.camel.builder.RouteBuilder;
import org.apache.camel.component.aws2.s3.AWS2S3Constants;

public class S3EvetListener extends RouteBuilder {
    @Override
    public void configure() throws Exception {
        from("aws2-s3://{{bucketName}}?deleteAfterRead=false")
            .routeId("s3-event-listener")
            .log("Received S3 event: ${header.CamelAwsS3EventType}")
            .process(exchange -> {
                String key = exchange.getIn().getHeader(AWS2S3Constants.KEY, String.class);
                String bucketName = "{{bucketName}}";

                // Log the S3 object key
                log.info("Processing file: " + key);

                // Download the file (the file content is in the message body)
                byte[] fileContent = exchange.getIn().getBody(byte[].class);
                log.info("Downloaded file content: " + new String(fileContent));
            });
    }
}

